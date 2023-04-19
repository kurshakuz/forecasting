export PWD=$(pwd)

# Build dependencies
sudo apt-get install gcc -y
sudo apt-get install g++ -y
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Download data
pip install gdown
sudo apt-get install unzip -y
gdown $DRIVE_LINK
unzip $PWD/ego4d_data_annot.zip -d /$PWD
rm $PWD/ego4d_data_annot.zip

# Clone thesis repository
test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd thesis-ws/
git pull

# Install requirements
pip install -r $PWD/thesis-ws/ego4d/forecasting/ego4d/requirements.txt

# Install slowfast
cd Ego4D-Future-Hand-Prediction
python3 slowfast_setup.py build develop

# Downlaod weights
mkdir -p $PWD/thesis-ego4d/results/output
mkdir -p $PWD/thesis-ego4d/results/checkpoints/
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl -P $PWD/thesis-ego4d/results

# Start training
python3 thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/tools/run_net.py --cfg $PWD/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50_colab.yaml OUTPUT_DIR $PWD/thesis-ego4d/results/
