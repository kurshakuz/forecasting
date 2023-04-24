# Start training


```bash
conda env create -f environment.yml
```

```bash
python3 setup.py build develop
```

Run the following command to start training.
```shell
python3 tools/run_net.py --cfg /home/dev/workspace/thesis-100doh-annotator/timesformer/configs/Ego4D/TimeSformer_divST_8x32_224_local.yaml OUTPUT_DIR /home/dev/workspace/thesis-ego4d/results/output/
```

```shell
python3 tools/run_net.py --cfg /home/dev/workspace/thesis-100doh-annotator/timesformer/configs/Ego4D/TimeSformer_divST_8x32_224.yaml OUTPUT_DIR /home/dev/workspace/thesis-ego4d/results/output/
```