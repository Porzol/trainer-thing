mkdir -p /mnt/vdb
mount /dev/vdb /mnt/vdb
mkdir -p dataset/
ln -s /mnt/vdb/Cam2 dataset/ 
ln -s /mnt/vdb/Cam4 dataset/ 
