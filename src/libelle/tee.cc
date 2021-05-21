/* Claudio Perez, 2021 */
/* tee.cc */
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
  return b * tf + (h - tf) *tw;
}

int main(int argc, char **argv){
  real_t b = 60.0, h=18.0, tw=18.0;
  real_t tf = 6.0;

  real_t A  = area(tf, tw, b, h);
  dual_t dA = area(dual_t(tf,1.0),tw,b,h);
  printd(dA);

  /* Alternative use case */
  // dual_t tf(6.0,1.0);
  // dual_t dA = b * tf + (h - tf) * tw;
  // printd(dA);  
}

