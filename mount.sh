mkdir -p /mnt/vdb
if ! mountpoint -q /mnt/vdb; then
    mount /dev/vdb /mnt/vdb
fi
mkdir -p dataset/
[ -L dataset/Cam2 ] || ln -s /mnt/vdb/Cam2 dataset/Cam2
[ -L dataset/Cam4 ] || ln -s /mnt/vdb/Cam4 dataset/Cam4
