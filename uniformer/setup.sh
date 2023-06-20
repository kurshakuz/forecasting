mkdir -p /workspace/thesis-ego4d/eccv-models/
wget https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_verb_uniformer_base_16x320_k600_ep9.pt -P ~/workspace/thesis-ego4d/eccv-models
wget https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_fhp_uniformer8x320.pth -P ~/workspace/thesis-ego4d/eccv-models
sudo apt-get install gcc -y
sudo apt-get install g++ -y
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Download data
pip install gdown
sudo apt-get install unzip -y


test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd /workspace/thesis-ws/
git pull
pip install -r /workspace/thesis-ws/eccv-ego4d/requirements.txt
cd /workpsace
gdown --no-cookies 1ozcEiH9g_utlwl60Gu3iHYnxnmf7slxP

unzip /workspace/thesis-ws/ego4d_data_annot.zip -d /workspace
