export PWD=$(pwd)

# Build dependencies
sudo apt-get install gcc -y
sudo apt-get install g++ -y
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Download data
pip install gdown
sudo apt-get install unzip -y
gdown "<link>"
unzip /workspace/ego4d_data_annot_w_test.zip -d /workspace
mv /workspace/ego4d_data_annot_w_test /workspace/ego4d_data_annot
# rm $PWD/ego4d_data_annot.zip

# Clone thesis repository
cd /workspace/
test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd /workspace/thesis-ws/
git pull

# Install requirements
pip install -r /workspace/thesis-ws/ego4d/forecasting/requirements.txt

# Install slowfast
cd /workspace/thesis-ws/uniformerv2
python3 setup.py build develop


pip install ftfy
python3 /workspace/thesis-ws/uniformerv2/extract_clip/extract_clip.py
# mv /workspace/thesis-ws/uniformerv2/vit_l14.pth /workspace/vit_l14.pth
mv /workspace/thesis-ws/uniformerv2/vit_b16.pth /workspace/vit_b16.pth

python3 /workspace/thesis-ws/uniformerv2/tools/run_net.py --cfg /workspace/thesis-ws/uniformerv2/exp/ego4d_hands/config.yaml
