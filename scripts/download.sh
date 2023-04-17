#bash

mkdir -p data

# USC-HAD
wget -c --timestamping  https://sipi.usc.edu/had/USC-HAD.zip -O data/USC-HAD.zip
unzip -o data/USC-HAD.zip -d data/

# ActRecTut
git clone git@github.com:andreas-bulling/ActRecTut
mv ActRecTut/Data/ data/ActRecTut
rm -rf ActRecTut

# ServerMachineDataset (SMD)
git clone git@github.com:NetManAIOps/OmniAnomaly data/OmniAnomaly
mv data/OmniAnomaly/ServerMachineDataset data/SMD && rm -rf data/OmniAnomaly