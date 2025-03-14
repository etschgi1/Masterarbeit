#!/bin/bash

export READLINK
READLINK="readlink"

if [ $(uname) == "Darwin" ]
then
  if which greadlink
  then
    READLINK="greadlink"
  else
    echo "install homebrew and then run:"
    echo "brew install coreutils"
    exit 1
  fi
fi

if [ -n "$1" ]
then
  mkdir -p "vsc-linux-intro_$1"
  WORKDIR="$(${READLINK} -f "vsc-linux-intro_$1")"
else
  echo
  echo "Please re-run and supply your name as an argument"
  echo
  echo "For example:"
  echo "./vsc-linux-intro.sh surname"
  echo
  exit 1
fi
pushd "${WORKDIR}" > /dev/null

echo "DeleteMeEmpty:" > README.txt
echo "try to delete this directory containing no files" >> README.txt
echo "" >> README.txt
echo 'p.s.: `man rmdir` is your friend' >> README.txt
echo "" >> README.txt
echo "" >> README.txt
echo "DeleteMeNonEmpty:" >> README.txt
echo "try to delete this directory containing some files" >> README.txt
echo "after deleting all files inside." >> README.txt
echo "" >> README.txt
echo 'p.s.: the `*`-pattern is your friend' >> README.txt
echo "" >> README.txt
echo "" >> README.txt
echo "DeleteMeRecursive:" >> README.txt
echo "try to delete this directory containing some files" >> README.txt
echo "without deleting any files by hand before deleting the directory." >> README.txt
echo "" >> README.txt
echo 'p.s.: `man rm` is your friend' >> README.txt

mkdir -p DeleteMeEmpty
mkdir -p DeleteMeNonEmpty
mkdir -p DeleteMeRecursive

pushd DeleteMeRecursive > /dev/null
for i in $(seq -w 100 110)
do
  echo "don't delete me by hand" > file.$i
done
popd > /dev/null

pushd DeleteMeNonEmpty > /dev/null
for i in $(seq -w 1 5)
do
  echo "delete me by hand" > file.00$i
done
popd > /dev/null

unpack () {
  if [ -n "$1" ]
  then
    echo "$1" | base64 -d - | tar xzf -
  fi
}

# AllNumbersFromTo
unpack "
H4sIAGgi3lkAA+3SvWrDMBAHcM96imviIVkaK/6CQtaOnfoCsiNjgSy1lg0pIe/ek+nUlC4N6fL/
DdadTjoOYWXtyzw0egzPox9e/WPok1vLWF2WcZV1KZdcFsWyRnmdJzKXVV3sZVZUSSbzMqsSym4+
yQ/mMKmRKGmUNb+f4ze6x0D3tX7YNcbtGhV6Idb0ZrUKmoKfx1bT1JtAnbGalDtyph3xmVZZGxPq
ZtdOxjsyjj74BoVeWxvb6JMauNUTh1+t1PV/xsXvuyRJZkJcbW+2dBZEg3GHVMZAnQ7pnoPOj+Tm
IU6QboJ+p3SIIde3XD16/hDptve0Ss988LJatp0WF/HfTw8AAAAAAAAAAAAAAAAAAAAAAPAnnwTb
qX0AKAAA"

# OneWordTwoStrings
unpack "
H4sIAMsh3lkAA+3Sz26bQBAGcM48xVfsU9TYJjG21JPfoZZ6XpuBXQt26P6R67fvkEpNE1U9RYlU
ze+yDHyaGQHs6WsKzvfHK3/j0MZVtMUb24h908xnvW/qp7rebp/OWbNrivqx3u23D5tdsys29eN+
2xTYvPUif5NjMgEoTmZw/85RiO+x0PtafFqfnF+fTLRluUDIHsm6iHgObkq4umRhIkbjbzChzyP5
FCUo926cMQ1kIsH4FvKC8D1zMsmx/4zEkuoD5wljHpKTJK7zDwbnE4M9/e63micfLcnocAPnJP04
WQroONwPzNO9JdPOUyQYieA62ZIQzfjcJcqyw4BAUcZJznm0rusoyLO56ZR/DaIfZpRlvsjlas2v
f3+Y0xkVdT2su1SoDCJadOirspRtcJn7Lu/KxXNx+KOolnfVi/JQlS2XAJ0tY3mRwlP50R9dKaWU
UkoppZRSSimllFJKKaWUUkop9V/4CVZc+pUAKAAA"

# FizzBuzz
unpack "
H4sIACPG3FkAA+3SQU+DMBgGYM79FZ+YJZAZoDIgkenBA7/CC47ONdnoUooHzP67XadTk8ltMzHv
c4C23/u1TaCSw/DYD0O08M4msfLZbP/mRcbdnB/mTpYXHk95XvBZaitewnme3XqUnO9KX/rO1JrI
e67XcjwndHeJC13WtWwX674R8840UkWrB8bimCqtNrQyZnsXx1p1wph6oRoRKf0SMyZbQ5tathS8
KtmE7I2RtV+VpRsulaZA0j3xkiTN7TtJ7Gg6DV35kHc9SwqubHJCPAvD4/LeVtsNbdmvPv5QPyyP
AbHuxLfmdKx3pO/XM0+ddzo5afwbkjbKfpYC/6n93GHnnlqYXreUlGzH2F9/dgAAAAAAAAAAAAAA
AAAAAAAA+AfeAU3KWNIAKAAA"

# Spacelog
unpack "
H4sIAA0j3lkAA+3SzY6CMBAHcM59ilk9L7ZA4bZPsgcLAiUiJbR1Y+LDW4iJnLzJXv6/y8w0/Zg0
Y0dV1b1pY6ujT+FBIeUcRSHFUossW+Is5XkkUpEXWSLSIo+4SLhMIuIf62jFW6cmoqhUffd+Xz3Z
LRra1v7rUHbDoVRWM7anqR77MBDUX0WSZpLUcAq5zNJE0F/ndFi4Od0NLd2Mp7471+FQM5kLOV2T
8W70jkxDx1ND3/rImDO+0mTnKYvDmDF2ub6qVxaX6syWM3SnNnRBu2cLv/fn+zv6Wd1TKbeq/vsX
AQAAAAAAAAAAAAAAAAAAAAAAtvcADnWe6QAoAAA="

# Spacelog2
unpack "
H4sIABAj3lkAA+3SsW7CMBAG4Mx+imtAYgGCEwekDpX6Bn0ABpzg4JQkjmIHhJq+exOEBJWqbsDy
f8vZlu98ks/WMlWF2YVzq707WfRWcTxEvor5ec+FOMeBWAqPR3y5EiGPhnM+RI8W92roVmudbIi8
RBb5//dUYx/R0GONXoIkr4JEWs3YiBpVF/1AUHHgYSRiktW2X8ciCjkdc6f7g5PTebWjk2mpyPeq
T8oaU5LTikzr6taRyWizzWimN4w506aa7DBl837MGCsP1911NU/knp1zqKNd3wX5lxbW3eV9n95u
6qTS3dTpyKot+bYLtClVcEntPhrzqVL33vm0ZvRLR/K4p8lX3eSVo/FySn7WKPXqT2ksvid/3E9N
0ZYVzRx79p8BAAAAAAAAAAAAAAAAAAAAAAAMfgB1PXDEACgAAA=="

# CatTogether
unpack "
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"

chmod +x *.sh

echo "Please copy the following line into your terminal and press enter:"
echo
echo "cd ${WORKDIR}"
echo
