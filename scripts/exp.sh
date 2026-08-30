brain_backbone="EEGProjectLayer"
vision_backbone="RN50"
dataset="meg"
seed=208000
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;

python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/meg/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

dataset="eeg"
seed=208000
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-05 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-06 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-07 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-08 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-09 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-10 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;

python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-05 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-06 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-07 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-08 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-09 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eeg/slvp.yaml --dataset $dataset --subjects sub-10 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

dataset="eegcvpr40"
seed=208000
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-05 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-06 --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;

python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-01 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-02 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-03 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-04 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-05 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
python main.py --config configs/eegcvpr40/slvp.yaml --dataset $dataset --subjects sub-06 --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;