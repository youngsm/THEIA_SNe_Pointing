import hashlib
import os
from pathlib import Path
from timeit import default_timer as timer
import json

import h5py
import numpy as np
import scipy.stats
from chroma.cache import Cache, GeometryNotFoundError
from chroma.event import Vertex
from chroma.loader import load_geometry_from_string
from chroma.log import logger
from tqdm import trange

import geometry.theia as theia
from brody.daq.simple_daq import simple_daq
from brody.misc_utils import (THEIA_HEIGHT_50KT, THEIA_RADIUS_50KT, random_pos,
                              random_three_vector, refractive_index_long,
                              refractive_index_short)
from brody.reconstruction import PromptDirectionStaged
from brody.unpack import Unpack

# NOTE: I copied over the eES_Gen class from 
# https://github.com/JamesJieranShen/eES_Gen
# to the `generator` folder
from generator.eEs_generator import eES_Gen

generator = eES_Gen()
FolderName = "2023-03-23_SNe"

def __configure__(db):
    
    db.pdf_input = (Path('/home/youngsam/portal/portal/sims/') \
        / "cuton_wavelength_study" \
        / "theia_knight-edmund_filters_dichroicon-perfect-inset-absorb-8-offset-close" \
        / "coord_460.pickle"
        ).as_posix()

    db.SNe_events = []

    # load fields from the custom_optics package
    db.logger = logger
    db.logger.setLevel("INFO")
    db.load_package("custom_optics")
    db.compress_cache = True

    db.chroma_photons_per_batch = 500_000
    db.chroam_max_steps = 1000
    db.chroma_g4_processes = 4
    db.chroma_keep_hits = False
    db.chroma_keep_photons_beg = False
    db.chroma_keep_photons_end = True
    db.chroma_keep_flat_hits = True

    db.chroma_particle_tracking = False
    db.chroma_photon_tracking = False

    db.theia_target = db.labppo_scintillator
    db.theia_pmt = "dichroicon-perfect-inset-absorb-8-offset-close"
    db.theia_coverage = 0.88
    db.theia_target_kilotons = 50.0
    db.theia_pmt_diameter_inches = 20.0
    db.theia_pmt_tts = 1.0  # PMT transit time spread (ns)
    db.theia_cuton = 460.0
    db.theia_towards_zero = False

    db.long_pc = db.extended_green_bialkali_photocathode
    db.short_pc = db.r7600_photocathode
    db.prompt_cut = 2.5  # prompt cut (ns) on hit time residuals to coordinate with

    db.notify_event = 50  # after x events notify with summary
    db.theta_C = None  # [deg]
    db.cone_length = None  # [mm]

    db.cache = Cache(compress=db.compress_cache)

    # filters to use
    db.dichroic_lp = db.KnightDichroicLongpassWater
    db.dichroic_sp = db.EdmundDichroicShortpassWater
    db.absorb_lp = db.KnightAbsorbingLongpassWater
    
    db.SNE_DIR = random_three_vector()


def __define_geometry__(db):
    geo_fmt = "theia-{}-{}-{}kt-{}cvg-{}nm-{}-{}-{}-{}-{}-{}"
    def filter_name(x): return getattr(x, "filename", x.name)
    geo_name = geo_fmt.format(
        db.theia_target.name,
        db.theia_pmt,
        db.theia_target_kilotons,
        db.theia_coverage,
        db.theia_cuton,
        "t0" if db.theia_towards_zero else "nt0",
        filter_name(db.dichroic_lp),
        filter_name(db.dichroic_sp),
        filter_name(db.absorb_lp),
        db.long_pc.name,
        db.short_pc.name
    )

    theta_C, cone_length = db.theta_C, db.cone_length

    geo_name += f"-{theta_C}"
    geo_name += f"-{cone_length}"

    hashed_geoname = hashlib.sha256(geo_name.encode()).hexdigest()

    try:
        print("Detector name", hashed_geoname)
        det = load_geometry_from_string(hashed_geoname, compressed=db.compress_cache)
        db.det = det
        db.logger.info("Loaded geometry from cache")
    except GeometryNotFoundError:
        db.logger.info("Building geometry")
        det = theia.build_detector(
            flat=True,
            pmt=db.theia_pmt,
            target=db.theia_target,
            det_radius=theia.det_radius(db.theia_target_kilotons),
            coverage=db.theia_coverage,
            rescale=db.theia_pmt_diameter_inches,
            towards_zero=db.theia_towards_zero,
            default_optics=db,
            cuton=db.theia_cuton,
            dichroic_lp=db.dichroic_lp,
            dichroic_sp=db.dichroic_sp,
            absorb_lp=db.absorb_lp,
            long_pc=db.long_pc,      # DETERMINE LONG PHOROCATHODE QE CURVE HERE
            short_pc=db.short_pc,    # DETERMINE SHORT PHOTOCATHODE QE
            theta_C=theta_C,
            maxL=cone_length,
            cache_bvh=True
        )
        db.cache.save_geometry(hashed_geoname, det)
        db.det = det
    return det

def __event_generator__(db):
    db.nueES_RATE_MEAN  = 478.7188           # from SNe_recon.ipynb
    db.nueES_RATE_SIGMA = 6.629323235444173

    # sample from a Gaussian distribution
    nueES_RATE = scipy.stats.norm.rvs(db.nueES_RATE_MEAN, db.nueES_RATE_SIGMA).astype(int)
    for _ in trange(nueES_RATE, desc="Generating supernova events", 
                    leave=False, ncols=80, position=2):
        event = generator.genEvent(db.SNE_DIR, eThreshold=1., nuThreshold=2.)

        db.SNe_events.append([event['flavor'],
                              event['nuEnergy'],
                              event['sn_direction'],
                              event['eKE'],
                              event['eDir']])

        yield Vertex("e-",
                     random_pos(THEIA_HEIGHT_50KT, THEIA_RADIUS_50KT),
                     event['eDir'],
                     event['eKE'],
                     t0=0)

def __simulation_start__(db):
    """Called at the start of the event loop"""
    db.ev_idx = 0
    db.t_sim_start = timer()
    
    db.t_sim_start = timer()

    db.out_dir = Path("/nfs/disk1/youngsam/sims").expanduser() / FolderName
    db.out_dir.mkdir(parents=True, exist_ok=True)
    print("created", db.out_dir)
    unpack_fmt = lambda x: f"unpack_{str(x).zfill(6)}.h5"
    
    db.SNe_number = 0
    while (db.out_dir / unpack_fmt(db.SNe_number)).exists():
        db.SNe_number += 1
    db.unpack_save_as = (db.out_dir / unpack_fmt(db.SNe_number)).as_posix()
    db.recon_save_as = (db.out_dir / f"recon_{str(db.SNe_number).zfill(6)}.h5").as_posix()

    coord = PromptDirectionStaged.Coordinators(db.pdf_input)
    db.dirfit_mask_l  = db.det.channel_index_to_channel_type == 2
    db.dirfit_mask_s  = db.det.channel_index_to_channel_type == 1
    db.group_velocity = [coord["long"].group_velocity, coord["short"].group_velocity]
    db.recon = PromptDirectionStaged(coord, db.prompt_cut, db.group_velocity)
    db.unpacker = Unpack(db.det, db.group_velocity, db.theia_pmt_tts, db.unpack_save_as)
    
    db.truth_vals  = []
    db.recon_vals  = []
    db.cosalphas1D = []

    
def __process_event__(db, ev):
    """Called for each generated event"""
    db.ev_idx += 1
    # -- unpack
    db.unpacker.digest_event(ev)
    
    # -- reconstruct
    true_pos = ev.vertices[0].pos
    true_time = ev.vertices[0].t0
    true_dir = ev.vertices[0].dir
    
    hit_channels, positions, times = simple_daq(
        ev, db.det.channel_index_to_position, tts=db.theia_pmt_tts)

    assert np.any(hit_channels)
    
    res = db.recon.fit(
        positions,
        times,
        db.dirfit_mask_l[hit_channels],
        db.dirfit_mask_s[hit_channels],
        truth_vals=np.concatenate([true_pos, [true_time], true_dir]),
        use_truth=False
    )
    if res.get('dir1D', None) is not None:
        db.truth_vals.append(np.concatenate(
            [true_pos, [true_time], true_dir]))
        db.recon_vals.append(np.concatenate(
            [res["pos+time"], res["dir1D"]]))
        cosalpha1D = np.dot(res['dir1D'], true_dir)
        db.cosalphas1D.append(cosalpha1D)
    else:
        db.truth_vals.append([-999]*7)
        db.recon_vals.append([-999]*7)
        db.cosalphas1D.append(-999)

    if len(db.cosalphas1D) > 0:
        try:
            transformed_cosalpha = [i if i != -999 else -1 for i in db.cosalphas1D]
            costheta_sorted1D = np.sort(np.arccos(transformed_cosalpha))
            costheta_sorted1D = costheta_sorted1D[~np.isnan(costheta_sorted1D)]
            sigma1D = np.degrees(
                costheta_sorted1D[int((0.68)*(len(costheta_sorted1D)-1))])
        except IndexError:
            sigma1D = 180
    else:
        sigma1D = 180

    if db.ev_idx != 0 and db.ev_idx % db.notify_event == 0:
        t_now = timer()
        print(" * simulated/reconstructed", db.ev_idx, "events at",
              (t_now - db.t_sim_start) / db.ev_idx, "s/ev")
        if db.chroma_keep_flat_hits:
            print("   - nhit", len(hit_channels), "avg time", np.mean(times))
        print("   - sigma cos[alpha] 1D       %8.3f" % np.mean(sigma1D))
        time_res = np.asarray(db.recon_vals)[:, 3]-np.asarray(db.truth_vals)[:, 3]
        print("   - time resolution 1D:       %8.3f ns" % time_res.mean())


def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.t_sim_end = timer()
    print("Ran %i events at %0.3f s/ev, which took\n %.3f min" %
          (db.ev_idx, (db.t_sim_end - db.t_sim_start) / db.ev_idx,
           (db.t_sim_end - db.t_sim_start) / 60,))
    
    # -- close unpacker
    db.unpacker.close()
    flavor_map = {flav: i for i, flav in enumerate(generator.flavors)}
    if not (db.out_dir / "flavor_map.json").exists():
        with open(db.out_dir / "flavor_map.json", "w") as f:
            json.dump(flavor_map, f, indent=4)
    
    # -- save results
    with h5py.File(db.recon_save_as, "w") as hf:
        hf["truth"] = list(db.truth_vals)
        hf["recon"] = list(db.recon_vals)
        hf['cosalpha'] = list(db.cosalphas1D)

        SNe_events = np.asarray(db.SNe_events)
        hf['flavor']        = list(map(lambda x: flavor_map[x], SNe_events[:, 0]))
        hf['nuEnergy']      = list(SNe_events[:, 1])
        hf['sn_direction']  = list(SNe_events[0, 2])  # all events have the same direction
        hf['eKE']           = list(SNe_events[:, 3])
        hf['eDir']          = list(SNe_events[:, 4])
