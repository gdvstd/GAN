NAME="cycleGAN_200epoch"
DATASET="pokemon-sprites"
CHANNELS=3


nohup python /root/PyTorch-GAN/implementations/cyclegan/cyclegan.py --experiment_name $NAME \
 --dataset_name $DATASET --channels $CHANNELS &> log.txt &