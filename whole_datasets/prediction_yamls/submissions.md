#NOTE: Activate plasmodesmata_dacapo
bsub -P cellmap -n 4 -o /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_3mb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_3mb.err python /groups/scicompsoft/home/ackermand/Programming/ml_experiments/scripts/submit.py predict -p 2025-02-15_3mb.yaml -w 100

bsub -P cellmap -n 4 -o /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_3rb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_3rb.err python /groups/scicompsoft/home/ackermand/Programming/ml_experiments/scripts/submit.py predict -p 2025-02-15_3rb.yaml -w 100

bsub -P cellmap -n 4 -o /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_2lb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/predictions/2025-02-15_2lb.err python /groups/scicompsoft/home/ackermand/Programming/ml_experiments/scripts/submit.py predict -p 2025-02-15_2lb.yaml -w 100
