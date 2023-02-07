
# RSNA breast cancer detection

# Data

## DICOM

Digital Imaging and Communications in Medicine（ダイコム）は医療用画像のフォーマット。python で DICOM を読み込むライブラリとして `pydicom` が開発されている。DICOM は画像としてのピクセル配列だけではなく、医療用のメタデータを保持している。

## Test data

> [train/test]_images/[patient_id]/[image_id].dcm The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

評価する private data では 8000人の患者のデータを取り扱う必要がある（〜32,000枚）。



# Train & Submit flow

## 新しいデータセットの作成

### データセットから作成する

1. データセットにしたいnotebook を開く
2. 「Data」のタブ選択、右の「Output」の横のハンバーガーメニューを選択
3. 「+New dataset」から選択しているnotebookの出力をデータセットにできる

## Train


### 学習時
1. コードをローカルで修正し、動作試験を実施
2. git push
3. Kaggle の学習用ノートブックを実行（[nb08_RSNA_train_v1](https://www.kaggle.com/code/kosuketakeda/nb08-rsna-train-png-v1)）
   1. 学習時は internet on なので、常に最新コードを git から pull してくるようにしている

### 学習終了 & 評価する時

4. 学習終了後、GCS に転送されている `mlruns` のディレクトリをローカルに持ってくる （`gsutil cp -r gs://kaggle-kabupen-rsna-bucket/mlruns .`）
5. ローカルで `mlflow ui` の実施、MLFlow 上で結果を眺める

## Submit 

このコンペでは Internet off でnotebookが走る必要があるので、
1. 最新版コードの整備 [rsnagit](https://www.kaggle.com/datasets/kosuketakeda/rsnagit)
2. モデルの整備 [rsnamodel](https://www.kaggle.com/datasets/kosuketakeda/rsnamodel)
3. ライブラリの整備 [rsnalibrary](https://www.kaggle.com/code/kosuketakeda/nb11-rsna-library)

を事前に行う必要がある。以上の事前準備をした段階で

- submitノートブックの編集 [nb10_RSNA_submit_v2](https://www.kaggle.com/code/kosuketakeda/nb10-rsna-submit-v2)
  - rsnagit の「Check for updates」の実行、最新版を使用する
  - Save
- ジョブ完了後、「Submit to Competition」の実行

### コードの整備

- データセットの[rsnagit](https://www.kaggle.com/datasets/kosuketakeda/rsnagit)へ飛び、ハンバーガーメニューから「Update」の選択。最新のgitコミットの反映

- ※注意事項
  - conda の事前 download はわからないので、condaでしか入らないライブラリは極力使用しないこと

### モデルの整備

- submit したいモデルをアップロードする
- rnsamodel へ飛び、「+New Version」からモデルをアップロードする

### ライブラリの整備

新しいライブラリを追加したときは、適宜 `rsnalibrary` を修正する必要があり（[Dataset/Your work/](https://www.kaggle.com/kosuketakeda/datasets?scroll=true)から確認できる）、修正用ノートブックを実行する。

1. https://www.kaggle.com/code/kosuketakeda/nb11-rsna-library へ飛ぶ
2. Edit、新しいライブラリの `pip download ...` を付け足す
3. Saveする
4. （Check）ジョブが終われば Dataset/rsnalibrary が自動で更新される

- ※注意事項
  - ライブラリによっては zip 形式で pip download されるが、データセットに入れた場合 Kaggle では自動的に解答されてしまう。そこで `.tmp`をつけてデータセットに入れ込む必要がある
  - `.tmp` は submit 用ノートブックで適切な処理を行う


# Log

## 2023/02/07

### train

- Adam, lr = 0.005 --> 0.001 に変更
- ReduceLROnPlateau、patience=3 に変更, mode="min" に変更（lossを見ているはずなので...）
- Augumentationの更新
  - CoarseDropout, RandomBrightness の追加

### submit  
- (!) モデルを取り違えて、50エポック学習したのに 9エポック目のモデルで sbumit していた
  - `model_fold1_epoch49_vacc0.545_vpfbeta0.504.pth` で再度submitしてみる

## 2023/02/06

- 1fold, 数エポックだけ学習したモデル（いちおう vfpobeta=0.5程度）をサブミットしてみたら、Score=0.3のままだった
  - 512x512 PNG 画像を使っているつもり
  - なぜだ？あまりにも乖離している
  - ひとまず学習を完全に終わらせるべく、downsampling を導入して 2000 vs 2000 サンプル程度の学習を試みてみる
  - ここから前処理にこだわってとりあえずスコアを伸ばしてみよう
- downsampling, 3fold, 15 エポック学習したモデルをサブミットすると、Score=0.4に上がった（[submitt]()）
  - 512x512 PNG画像
  - やはりスコア0.3は学習モデルが汎用的なものになっていなかったのだな？

## 2023/02/02

- ついに整ったのでここから本番
- しかし training は全然学習が進んでいないことがわかった（なぜ）。
- [メトリックについて](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369886)
- この[ベースライン](https://www.kaggle.com/code/theoviel/rsna-breast-baseline-inference)をもとに修正する

## 2023/01/29

- submit.py が timeout する理由がわかった気がする。適当な数字を入れてsubmit すると（submit.pyを実行せず）submit が sucess になった。
  - つまり、DICOM --> PNG がテストデータに走る際に、これは自分の知っているテストデータではなく運営が持っている private データでありその量は不明
  - この処理に大量に時間がかかってしまっているのだと思う（submission.csvのフォーマットエラーだと他のエラーメッセージになる様子）
- 渡されている dicom は何らかの前処理がかかった後であり、それは site_id, machine_id ごとに preprocessing は異なっているはず

### dicom2png の高速化

- https://www.kaggle.com/code/remekkinas/fast-dicom-processing-1-6-2x-faster?scriptVersionId=113360473
- https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/371033#2060484


- dicomsdl の pip install @mac
  - `mach-o file, but is an incompatible architecture (have (x86_64), need (arm64e)))` とエラーが出る。dicomsdl は Intelアーキテクチャ用らしく、M1 macではそのままでは動作しないっぽい
  - arch: posix_spawnp: python: Bad CPU type in executable 
  - ロゼッタのインストールしても無駄
  - 結局、
```
$ git clone --recursive https://github.com/tsangel/dicomsdl
$ python setup.py install
```
の実施した。

## 2023/01/22

- PNG へ変換したデータセットの準備と、それに対応したコードの修正
- 

## 2023/01/21

- `!pip install -qU pylibjpeg pylibjpeg-openjpeg pylibjpeg-libjpeg pydicom python-gdcm` をまずはじめに実行する必要がある（そうでないとエラーが出る。notebookならリロードが必要）
- mlflow, hydra に対応させた
- Kaggle 環境のmlflow のインストールでつまづいたが、pip ではなく conda で入れることでどうにか使えた。ただし Kaggle カーネルが提供しているpython は3.7 なので要注意。