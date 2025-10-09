mkdir -p /mnt/vdb
if ! mountpoint -q /mnt/vdb; then
    mount /dev/vdb /mnt/vdb
fi
mkdir -p dataset/
ln -s /mnt/vdb/Cam2 dataset/ 
ln -s /mnt/vdb/Cam4 dataset/ 
