mkdir ext
git clone https://github.com/GuyTevet/motion-diffusion-model.git ext/mdm
git clone https://github.com/EricGuo5513/HumanML3D.git ext/HumanML3D

cd ext/mdm
git checkout af061ca7c7077fb144c0094a5a72932b967647b6

cd ../..



unzip ./ext/HumanML3D/HumanML3D/texts.zip -d ./ext/HumanML3D/HumanML3D/
cp -r ext/HumanML3D/HumanML3D ext/mdm/dataset/HumanML3D
rm -rf ext/HumanML3D

wget https://github.com/Asixa/RF-Genesis/releases/download/v1.0.0/RFGen_Dependencies.zip

unzip RFGen_Dependencies.zip
unzip RFGen_Dependencies/glove.zip -d ./ext/mdm/
unzip RFGen_Dependencies/smpl.zip -d ./ext/mdm/body_models/
unzip RFGen_Dependencies/t2m.zip -d ./ext/mdm/
unzip RFGen_Dependencies/kit.zip -d ./ext/mdm/

wget https://github.com/Asixa/RF-Genesis/releases/download/v1.1.0/RFLoRA.zip

unzip RFLoRA.zip -d ./models


unzip RFGen_Dependencies/smpl_basicModel_v1.0.0.zip -d ./models/smpl_models
unzip RFGen_Dependencies/smpl_ply.zip -d ./models/



mkdir ./ext/mdm/save
unzip RFGen_Dependencies/humanml_trans_enc_512.zip -d ./ext/mdm/save/

mv RFGen_Dependencies/generate_rfgen.py ext/mdm/sample/generate_rfgen.py

rm -rf RFGen_Dependencies

echo "RFGen Setup completed successfully."