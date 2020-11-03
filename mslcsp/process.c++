#include <fstream.h>
#include <strstream.h>
#include<iomanip.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

#define MAXTOPICS 1000
int ntopics;
char topics[MAXTOPICS][20];
float val[MAXTOPICS][500]; char buf[41920],name[128]; int i,j,n;
void main(void) {
  ifstream is("allfiles.res");ofstream os("process.txt");
  is. getline(buf,sizeof(buf)); if(!is) return; istrstream readstr(buf);
  for(i=0;i<MAXTOPICS;i++)
    if(!(readstr>>topics[i])) break;
  ntopics=i;
  os<<setw(14)<<topics[0]; n=1;
  for(j=1;is&&j<500;j++) {{
    is. getline(buf,sizeof(buf)); if(!is) break; istrstream readstr(buf);
    readstr>>setw(sizeof(name))>>name;
    val[0][j] = strlen(name)+1;
    os<<setw(MAX(14,val[0][j]+3))<<name;
    for(i=1;i<ntopics;i++) readstr >> val[i][j];
    if(is) n++;
 }} os<<'\n'; os.precision(3);
  for(i=1;i<ntopics;i++) {
   os<<setw(14)<<topics[i];
   for(j=1;j<n;j++) 
    os<<" & "<<setw(MAX(11,int(val[0][j])))<<val[i][j];
    os<<'\n';
  }
}
