# ia_racing_imt
...
python train.py --algo tqc --env donkey-generated-track-v0 --eval-freq -1 --save-freq 20000
python train.py --algo tqc --env donkey-generated-track-v0 -i logs/tqc/donkey-generated-track-v0_2/rl_model_40000_steps.zip -n 5000
python enjoy.py --algo tqc --env donkey-generated-track-v0 -f logs/ --exp-id 7 --load-last-checkpoint
python enjoy.py --algo tqc --env donkey-generated-track-v0 -f logs/ --exp-id 7 --load-best
