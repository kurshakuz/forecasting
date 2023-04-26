export PWD=$(pwd)

# Build dependencies
sudo apt-get install gcc -y
sudo apt-get install g++ -y
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Download data
pip install gdown
sudo apt-get install unzip -y
gdown "$DRIVE_LINK&confirm=t"
unzip /workspace/ego4d_data_annot.zip -d /workspace
# rm $PWD/ego4d_data_annot.zip

# Clone thesis repository
cd /workspace/
test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd /workspace/thesis-ws/
git pull

# Install requirements
pip install -r /workspace/thesis-ws/ego4d/forecasting/requirements.txt

# Install slowfast
cd /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction
python3 slowfast_setup.py build develop

# Downlaod weights
mkdir -p /workspace/thesis-ego4d/results/output
mkdir -p /workspace/thesis-ego4d/results/checkpoints/
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl -P /workspace/thesis-ego4d/results

# Start training
python3 /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/tools/run_net.py --cfg /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50_vast_3090.yaml OUTPUT_DIR /workspace/thesis-ego4d/results/
