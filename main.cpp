#include <iostream>
#include <cmath>
using namespace std;

int main() {

    int r,w,l; 
    int cnt=1;

    while(true){

        cin>>r;
        if(r==0) break; 

        cin>>w>>l;

        if(2*r>=sqrt(w*w+l*l)){
            cout<<"Pizza "<<cnt<<" fits on the table"<<endl;
        }
        else
            cout<<"Pizza "<<cnt<<" does not fit on the table"<<endl;

        cnt++;
        
    } 
    
    return 0;

}

// 왜 재귀로 풀면 시간 초과가 나고 반복문으로 풀면 시간초과가 안나지??