# Data

`samples/` contains compact preprocessed arrays recovered from the historical
REC-NN workspace:

- `AIG_0.01.npy`: artificial irregular geometry example.
- `circular.npy`: circular phantom example.
- `tumor_4mm.npy`: 4 mm tumor examples.

Each array uses the channel convention documented in the project README.
These files are suitable for smoke tests and format inspection, but they are
not the complete AIG, DHH, Ella, or tumor datasets used in the papers.

Before publishing the repository, confirm that you have permission to
redistribute each dataset. Put non-public full datasets under `data/private/`
or `data/full/`; both paths are ignored by Git.
