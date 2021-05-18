#include "ad.cc"


template<typename TX, typename TC>
auto f(TX x, TC c){
    return c - x;
}

template<typename TX,typename TA, typename TB>
auto g(TX x, TA a, TB b){
  return a * x + f(x)*b;
}

template<typename TF,typename TW, typename TB, typename TH>
auto area(TF tf, TW tw, TB b, TH h){
  printd(b*tf);
  printd(h - tf);
  return b * tf + (h - tf) *tw;
}


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
  real_t A = area(tf, tw, b, h);
  printd(dA);
}
