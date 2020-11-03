//
#include <fstream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>

#define MN 1024
char fln[256];
int ll[MN],LL[MN],bb[MN],BB[MN],l,b,L,B,l1,l2,b1,b2,
 m,M,mm,MM,n,N,
 l1s,l1t,l2s,l2t,l1step,l2step,L1,L2,B1,B2,Lstep;
double fi;
double rnd1(void) { return double(rand())/RAND_MAX; }
void main(int argc,char **argv) {
 if(!(ifstream(argc>1 ? argv[1] : "cspgen.cfg")>>N>>m>>l1s>>l1t>>l1step
  >>l2s>>l2t>>l2step>>b1>>b2
  >>M>>L1>>L2>>Lstep>>B1>>B2>>fi))
   { perror("Please ensure cspgen.cfg or argv[1] as paramfile"); exit(1); }
 int i,j;
 for(l1=l1s;l1<=l1t;l1+=l1step)
 for(l2=l2s;l2<=l2t;l2+=l2step)
 if(l1<l2) {
  ostrstream(fln,sizeof(fln))
   <<'m'<<m
   <<'M'<<M
   <<'l'<<double(l1)/L2*1000<<'_'
   <<double(l2)/L2*100
   <<'b'<<b1<<'_'<<b2
   <<'L'<<L1
   <<'g'<<Lstep
   <<'B'<<B1<<'_'<<B2
   <<ends;
  ofstream os(fln); srand(561);
  os<<"Class parameters ("<<N<<" instances):\n";
  os<<"m="<<m<<" l1s="<<l1s<<" l1t="<<l1t<<" l1step="<<l1step
  <<" l2s="<<l2s<<" l2t="<<l2t<<" l2step="<<l2step<<" b1="<<b1<<" b2="<<b2
  <<"\nM="<<M<<" L1="<<L1<<" L2="<<L2<<" Lstep="<<Lstep
  <<" B1="<<B1<<" B2="<<B2<<" fi="<<fi<<'\n';
 for(n=1;n<=N;n++) {
  mm=0;
  for(i=1;i<=m;i++) {
   l=int(rnd1()*(l2-l1)+l1);
   b=int(rnd1()*(b2-b1)+b1);
   for(j=1;j<i;j++)
   if(ll[j]==l) { bb[j]+=b; l=0; break; }
   if(l!=0) { mm++; ll[mm]=l; bb[mm]=b; }
  }
  MM=1; LL[1]=L2; BB[1]=int(rnd1()*(B2-B1)+B1);
  for(i=2;i<=M;i++) {
   L=int(rnd1()*((L2-L1)/Lstep))*Lstep+L1;
   B=int(rnd1()*(B2-B1)+B1);
   for(j=1;j<i;j++)
   if(LL[j]==L) { BB[j]+=B; L=0; break; }
   if(L!=0) { MM++; LL[MM]=L; BB[MM]=B; }
  }
  os<<"\nNN="<<n<<"\n-1 "<<mm<<' '<<MM<<'\n';
  for(i=1;i<=mm;i++) os<<ll[i]<<' '<<bb[i]<<'\n';
  os<<'\n';
  for(i=1;i<=MM;i++) os<<LL[i]<<' '<<BB[i]<<'\n';
 } //:for n
 } //:if l1<l2
}

