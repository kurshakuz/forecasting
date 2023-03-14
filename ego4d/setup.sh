export AWS_ACCESS_KEY_ID="AKIATEEVKTGZNURSIMNK"
export AWS_SECRET_ACCESS_KEY="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"

# Set up the AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip >/dev/null
sudo ./aws/install >/dev/null 2>&1
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" && aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
rm "awscliv2.zip"

# Download the Ego4D Annotations to ego4d_data/
ego4d --output_directory="~/workspace/content/ego4d_data/" --datasets annotations full_scale --video_uid_file ./fho_hands_train_contact_uids.txt --benchmarks FHO -y
# ego4d --output_directory="~/workspace/content/ego4d_data/" --datasets annotations full_scale --video_uids 98dc31a1-edbb-42d6-8d1c-0b43fe6f9c28 --benchmarks FHO -y

# ls /workspaces/content/ego4d_data/v1/annotations | grep fho