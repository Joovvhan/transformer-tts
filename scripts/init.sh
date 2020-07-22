git submodule init
git submodule update

# https://gldmg.tistory.com/144
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1PxMubqKAYaDIsc3c-tcH_eXFrj_Znf6V" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1PxMubqKAYaDIsc3c-tcH_eXFrj_Znf6V" -o 19829_1011266_bundle_archive.zip
rm cookie

unzip 19829_1011266_bundle_archive.zip -d ./korean-single-speaker-speech-dataset/
rm 19829_1011266_bundle_archive.zip

cd waveglow
git submodule init
git submodule update

# https://gldmg.tistory.com/144
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF" -o waveglow_256channels_universal_v5.pt
rm cookie

cd ..

