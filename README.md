# THEIA_SNe_Pointing

![recon_000000](images/recon_000000.jpg)

## Files:

* `reconstruct_supernovae.py`: pyrat macro for simulating SNe in a 50 kT THEIA.
* `simrunner.py`: python script for running the macro on multiples GPUs and nodes on nubar.
* `SNe_recon.ipynb`: jupyter notebook for a cursory look at the results.

Actual reconstruction data for all 500 SNe's are not included in this repo. They are available on
nubar at `"/nfs/disk1/youngsam/sims/2023-03-23_SNe/`. For each SNe there are two files:
`unpack_<number>.h5` and `recon_<number>.h5`. The former is data from `brody`'s unpacker
module, the latter is the salient data saved in the macro:

I.e.,

```sh
/nfs/disk1/youngsam/sims/2023-03-23_SNe/recon_000000.h5
    cosalpha: 483 entries, dtype: float64
    eDir: 483 × 3 entries, dtype: float64
    eKE: 483 entries, dtype: float64
    flavor: 483 entries, dtype: int64
    nuEnergy: 483 entries, dtype: float64
    recon: 483 × 7 entries, dtype: float64
    sn_direction: 3 entries, dtype: float64
    truth: 483 × 7 entries, dtype: float64
```

The `flavor` dataset is the neutrino flavor as an enum. Specific mappings between the enum
and the true flavor is in the `flavor_map.json` file:

```json
{
    "nue": 0,
    "nuebar": 1,
    "numu": 2,
    "numubar": 3,
    "nutau": 4,
    "nutaubar": 5
}
```