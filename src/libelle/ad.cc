/* Claudio Perez
 *
 *
 */

#include <cstdio>

typedef double real_t;

class dual_t {
  public:
    real_t  real;
    real_t  dual;
    dual_t *next=NULL;
    dual_t operator + (dual_t b){
        return (dual_t) {
            .real = real + b.real,
            .dual = dual + b.dual
        };
    }
    dual_t operator + (real_t b){
        return (dual_t){
            .real = real + b,
            .dual = dual
        };
    }
    dual_t operator - (dual_t b){
        return (dual_t){
            .real = real - b.real,
            .dual = dual - b.dual
        };
    }
    dual_t operator - (real_t b){
        return (dual_t){
            .real = real - b,
            .dual = dual
        };
    }


    dual_t operator * (dual_t b){
        return (dual_t) {
            .real = b.real * real,
            .dual = b.dual * real + b.real*dual
        };
    }
    dual_t operator * (real_t b){
        return (dual_t) {
            .real = b * real,
            .dual = b * dual
        };
    }
};

void printd(dual_t y){
    printf("%f + %f\n",y.real,y.dual);
}

int main(int argc, char **argv){
    dual_t a = {3.0, 0.0};
    dual_t b = {2.0, 0.0};
    dual_t x = {4.0, 1.0};

    dual_t y = x*3.0 + b;

    printd(x*3.0 + b);
    printd(a*x + b);

    return 0;
}

