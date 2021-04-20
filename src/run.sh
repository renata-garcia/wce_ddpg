scp ../*txt fwc61:~/; nano script.sh ; rm ../*txt; git checkout ../; git pull; git status; ls ..
scp fwc61:~/run.txt .
cat run.txt >> script.sh
cat script.sh
rm run.txt
