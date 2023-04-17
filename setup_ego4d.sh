export PWD=$(pwd)

# Clone thesis repository
test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd thesis-ws/
git pull

# Install requirements
cd forecasting/
pip install -r requirements.txt

cd Ego4D-Future-Hand-Prediction
python3 slowfast_setup.py build develop

# Download data
cd $PWD
pip install gdown
gdown $DRIVE_LINK
unzip ego4d_data_annot.zip -d /$PWD
rm ego4d_data_annot.zip

# Downlaod weights
cd $PWD
mkdir -p /$PWD/thesis-ego4d/results/output
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl -P /$PWD/thesis-ego4d/results

# Start training
# python3 thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/tools/run_net.py --cfg /$PWD/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50_colab.yaml OUTPUT_DIR /$PWD/thesis-ego4d/results/
