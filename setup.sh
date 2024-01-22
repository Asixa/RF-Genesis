mkdir ext
git clone https://github.com/GuyTevet/motion-diffusion-model.git ext/mdm
git clone https://github.com/EricGuo5513/HumanML3D.git ext/HumanML3D

unzip ./ext/HumanML3D/HumanML3D/texts.zip -d ./ext/HumanML3D/HumanML3D/
cp -r ext/HumanML3D/HumanML3D ext/mdm/dataset/HumanML3D
rm -rf ext/HumanML3D

unzip RFGen_MDM_Dependencies.zip
unzip RFGen_MDM_Dependencies/glove.zip -d ./ext/mdm/
unzip RFGen_MDM_Dependencies/smpl.zip -d ./ext/mdm/body_models/
unzip RFGen_MDM_Dependencies/t2m.zip -d ./ext/mdm/
unzip RFGen_MDM_Dependencies/kit.zip -d ./ext/mdm/


unzip RFGen_MDM_Dependencies/smpl_baiscModel_v1.0.0.zip -d ./models/
unzip RFGen_MDM_Dependencies/smpl_ply.zip -d ./models/



mkdir ./ext/mdm/save
unzip RFGen_MDM_Dependencies/humanml_trans_enc_512.zip -d ./ext/mdm/save/

mv RFGen_MDM_Dependencies/generate_rfgen.py ext/mdm/sample/generate_rfgen.py

rm -rf RFGen_MDM_Dependencies

echo "RFGen Setup completed successfully."