START=$(date +%s)
##########################################################################
echo && echo " #################################################" 
echo " ##              SYNC THE JETSON NANO            " 
echo " #################################################" && echo  


echo && echo  " ---> Sending  files to the Jetson..." 
rsync -azP artifacts/inference  jetson@10.0.0.163:/home/jetson/Desktop/nvidia-inference  


echo && echo  " ---> Receiving files from Jetson..." 
rsync -azP  jetson@10.0.0.163:/home/jetson/Desktop/nvidia-inference   artifacts/inference

# #------------- CLEAN THE CODE -------------
# ./bash/run/run_clean_build_files.sh

