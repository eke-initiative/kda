# Knowledge Domain Annotation

**Requirements**

- maven
- Java 11+
- python3
- conda

To run the experiment:

1. Download the dataset from this link [TO_ADD]()
2. Unpack the dataset:

```
tar -zxvf KDA_dataset.tar.gz
```
3. Unpack virtual text documents

```
cd KDA_dataset/VirtualTextDocuments
tar -zxvf TPB.tar.gz
tar -zxvf LOV.tar.gz
tar -zxvf Laundromat.tar.gz
```

alternatively, you can generate virtual text documents by using [vdg](https://github.com/empirical-knowledge-engineering/vdg) (in this case Apache Maven is required to be installed on your machine). To do that:

a. Unpack input rdf files

```
cd KDA_dataset/InputRDF
tar -zxvf TPB.tar.gz
tar -zxvf lov.nq.tar.gz
tar -zxvf LOD_Laundromat.tar.gz
tar -zxvf labelMap.tar.gz
```

b. Download and run vdg

```
git clone https://github.com/empirical-knowledge-engineering/vdg.git
cd vdg/
mvn clean install
mkdir ../VirtualTextDocuments/Laundromat
mvn exec:java  -Dexec.cleanupDaemonThreads=false -Dexec.mainClass="it.cnr.istc.stlab.lgu.Main"  -Dexec.args="Laundromat ../Laundromat ../labelMap ../VirtualTextDocuments/Laundromat"  -DjvmArgs="-Xmx32g"
mkdir ../VirtualTextDocuments/TPB
mvn exec:java  -Dexec.cleanupDaemonThreads=false -Dexec.mainClass="it.cnr.istc.stlab.lgu.Main"  -Dexec.args="TPB ../TPB ../dataset_ids ../VirtualTextDocuments/TPB"  -DjvmArgs="-Xmx32g"
mkdir ../VirtualTextDocuments/LOV
mvn exec:java  -Dexec.cleanupDaemonThreads=false -Dexec.mainClass="it.cnr.istc.stlab.lgu.Main"  -Dexec.args="LOV ../lov.nq.tar.gz ../VirtualTextDocuments/LOV"  -DjvmArgs="-Xmx32g"
```


4. Preprocess virtual documents:

```
git clone https://github.com/empirical-knowledge-engineering/kda.git
cd kda/
conda create --name kda --file requirements.txt
conda activate kda
python preprocess_virtual_documents.py <KDA_Dataset_path>
```

Alternatively, you can use preprocessed dataset available in ``KDA_Dataset/experiment``

5. Create stratified folds

```
python create_folds.py <KDA_Dataset_path>
```

Pre-computed folds are available in ``KDA_Datasets/experiment``

6. Resample folds with MLSMOTE (from directory ``KDA_Dataset/``):

```
python resample_folds.py <KDA_Dataset_path>
```

Resampled folds are available in ``KDA_Datasets/experiment``


7. Train and test classifier

```
python train_and_test_classifier.py  <KDA_Dataset_path>
