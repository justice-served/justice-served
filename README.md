# justice-served
NLP Project: binary classification of criticality for Swiss legal cases (i.e. likelihood of a case being referred to the supreme court)

## Uploading to GCloud VM

In order to upload this project to a GCloud VM, follow these steps (WARNING: the following step can overwrite files
already on the VM):

- install `gcloud` client
- authenticate to Google using AllenNLP hacks account (see Discord)
- `cd` to this directory
- `gcloud compute scp --recurse * connor@justice-served-2:~ --zone "us-east1-b" --project "maximal-copilot-324819"`
- This will upload the contents of the local working directory to `/home/connor` on the VM

You can then connect to the VM via SSH with the following command:

```bash
gcloud beta compute ssh --zone "us-east1-b" "justice-served-2"  --project "maximal-copilot-324819"
```

## Unloading the Data

The Criticality data are found in the `text.zip` file shared over telegram by Joel. Unzip that file and place it in
the `adapter_script/data/` directory.

## Running the Script

Here are the dependencies:

```bash
conda install -c pytorch pytorch cpuonly
pip install -r adapter_script/requirements.txt
pip install -U adapter-transformers
```

I've been trying to run like this:

```bash
cd adapter_script
./run.sh --model_name="Musixmatch/umberto-commoncrawl-cased-v1" --type="hierarchical" --language="it" --train_language="it" --mode="train" --sub_datasets=False --seed=123 --debug=True
```
