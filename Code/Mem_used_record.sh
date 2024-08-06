#!/bin/bash

Mem_max=0
Swp_max=0

while ps -p $1 > /dev/null;
do
    Mem_now=$(free -m | awk 'NR==2{printf $3}')
    Swp_now=$(free -m | awk 'NR==3{printf $3}')
    refresh_tag=0

    if [ $Mem_now -gt $Mem_max ]; then
        Mem_max=$Mem_now
        refresh_tag=1
    fi

    if [ $Swp_now -gt $Swp_max ]; then
        Swp_max=$Swp_now
        refresh_tag=1
    fi

    if [ $refresh_tag -eq 1 ]; then
        echo "PID : $1" > Mem_Record.log
        printf "Max Used Mem : %.3f\n" $(echo "scale=3; $Mem_max/1024" | bc) >> Mem_Record.log
        printf "Max Used Swp : %.3f\n" $(echo "scale=3; $Swp_max/1024" | bc) >> Mem_Record.log
    fi
    # printf "Max Used Mem : %.3f" $Mem_max/1024 >> Mem_Record.log
    # printf "Max Used Swp : %.3f" $Swp_max/1024 >> Mem_Record.log
    # echo "Max Used Mem : $Mem_max" >> Mem_Record.log
    # echo "Max Used Swp : $Swp_max" >> Mem_Record.log
    sleep 1
done