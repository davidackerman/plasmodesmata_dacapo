# NOTE: conda activate rusty_mws
bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/mws/2025-02-15_3m.out -e /nrs/cellmap/ackermand/logs/mws/2025-02-15_3m.err python 2025-02-15_3m.py
bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/mws/2025-02-15_3r.out -e /nrs/cellmap/ackermand/logs/mws/2025-02-15_3r.err python 2025-02-15_3r.py
bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/mws/2025-02-15_2l.out -e /nrs/cellmap/ackermand/logs/mws/2025-02-15_2l.err python 2025-02-15_2l.py

bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_3mb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_3mb.err python 2025-02-15_3mb.py
bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_3rb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_3rb.err python 2025-02-15_3rb.py
bsub -P cellmap -n 64 -o /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_2lb.out -e /nrs/cellmap/ackermand/logs/whole_dataset/mws/2025-02-15_2lb.err python 2025-02-15_2lb.py