* paste a few function declarations (maybe even with comments !) into opencv/Core.hpp
* run python gen_go.py ../src/gocv.io/x/gocv  headers.txt
* run go test gocv.io\x\gocv
* fix testcases ;)
* merge generated code into gocv codebase (copy/paste)

it's ugly as shit, but already saves a lot of copy/paste issues
