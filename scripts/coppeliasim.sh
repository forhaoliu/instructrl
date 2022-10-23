#! /bin/bash

# Install dependencies
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3

# Install CoppeliaSim
OS_Version="$(lsb_release -sr | awk -F. '{print $0}')"
OS_NAME=$( echo ${OS_Version} | tr '.' '_' )
if [ "$OS_NAME" != '18_04' ] ; then
    OS_NAME='20_04'
fi
export COPPELIASIM_PREFIX=$HOME
if [ ! -r $COPPELIASIM_PREFIX/CoppeliaSim_Edu_V4_1_0_Ubuntu$OS_NAME ] ; then
    wget -P /tmp https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu$OS_NAME.tar.xz --no-check-certificate
    tar -xf /tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu$OS_NAME.tar.xz -C $COPPELIASIM_PREFIX
    rm /tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu$OS_NAME.tar.xz
fi

# Update LD_LIBRARY_PATH
# rm -rf $HOME/.bashrc
cat > $HOME/.bashrc <<- EndOfFile
export COPPELIASIM_ROOT=\$COPPELIASIM_PREFIX/CoppeliaSim_Edu_V4_1_0_Ubuntu\$OS_NAME
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT
EndOfFile
source $HOME/.bashrc

# Headless GPU render
sudo apt-get -y install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
if [ "$OS_NAME" == '18_04' ] ; then
    wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate
else
    wget https://sourceforge.net/projects/virtualgl/files/3.0.1/virtualgl_3.0.1_amd64.deb/download -O virtualgl_3.0.1_amd64.deb --no-check-certificate
fi
sudo dpkg -i virtualgl*.deb
rm virtualgl*.deb
