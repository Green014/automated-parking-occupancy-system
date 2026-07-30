# Stage N formal parking-data candidate audit

Decision: **blocked**. No new source currently satisfies every formal
parking-occupancy requirement, so OS0 and T0-T3 were not run.

## Local candidates

VIRAT 0502 is already consumed development evidence and cannot be presented
as a new test. Grand Bassin contains one fixed-looking aerial camera split
into several clips, machine-generated vehicle boxes, and only incomplete
occupied-bus-bay truth; it has no frozen vacant intervals or transition-frame
truth. Local PKLot and CNRPark-EXT assets are sparse occupancy images rather
than continuous event videos, and both have already influenced development.
NDISPark is detector-training material without frozen slot polygons or
transition truth.

## Public candidates checked

- [CNRPark+EXT](https://cnrpark.it/) has nine cameras but one physical parking
  lot, with 4,081 sparse frames and patch-level free/busy labels. It cannot
  provide physical development/test scene isolation or transition frames.
- The [official PKLot description](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
  covers different parking lots and fixed views, but uses a five-minute
  time-lapse. It is not continuous enough to adjudicate entry, departure,
  pass-by, and exact transition frames, and the local copy is already
  consumed.
- [Dragon Lake Parking](https://zenodo.org/records/10084683) contains parking
  manoeuvres and rich tracks, but uses a flying drone over one parking lot;
  it is not a fixed-camera, two-physical-scene occupancy test.
- [ACPDS](https://arxiv.org/abs/2107.12207) separates parking lots across
  splits, but intentionally captures each image from a unique view. It is an
  image classifier dataset, not a fixed-camera event sequence.
- [UFPArk](https://www.researchgate.net/publication/349567141_Public_Dataset_of_Parking_Lot_Videos_for_Computational_Vision_Applied_to_Surveillance)
  provides surveillance video clips, but the audited public material did not
  establish a standard reusable dataset licence, per-slot interval truth, or
  physically isolated development/test scenes.
- The [AGH Parking Database](https://qoe.agh.edu.pl/parking-database/) offers
  camera videos and licence-plate-coordinate truth, not parking-slot
  occupancy polygons and intervals; its page requests citation but does not
  state a standard dataset licence.

## Gate outcome

The next admissible dataset must first freeze two or more physically distinct
fixed-camera scenes, an explicit licence, scene-disjoint development/test
roles, slot polygons, and human-reviewed interval or transition truth. Only
then may development exercise the interfaces; after configuration freeze,
test may run once. LMOT cannot fill this gate, and VIRAT 0502 will not be
reused as a nominal new test.
