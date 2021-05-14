#include "ad.cc"

// #define JACFWD(FUNC, T1, T2, T3) FUNC<decltype(A1), decltype(A2)>(A1,A2)




template<typename TF,typename TW, typename TB, typename TH>
auto area(TF tf, TW tw, TB b, TH h){
  printd(b*tf);
  printd(h - tf);
  return b * tf + (h - tf) *tw;
}

// template<TF, TW, TB, TH>
// T moment(T b, real_t h){

int main(int argc, char **argv){
  real_t E = 29000000.0;
  real_t b = 60.0, h=18.0, tw=18.0;
/*
  dual_t tf(6.0,1.0);
  dual_t dA = b * tf + (h - tf) * tw;
  printd(dA);
*/
  real_t tf = 6.0;
  dual_t dA = area(D(tf),tw,b,h);
  printd(dA);
}
