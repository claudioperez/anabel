/* Claudio Perez, 2021 */
/* ad.cc */
#include <cstdio>
#define D(x) dual_t((x),1.0)
typedef double real_t;

struct dual_t {
    real_t  real;
    real_t  dual = 0.0;
    /* First constructor */
    dual_t (real_t a){
        real = a;
        dual = 0.0;
    }
    /* Second constructor */
    dual_t (real_t a, real_t b){
        real = a;
        dual = b;
    }

    dual_t operator + (dual_t const & b){
        return (dual_t) {
            .real = real + b.real,
            .dual = dual + b.dual
        };
    }

    dual_t operator + (real_t const & b){
        return (dual_t){
            .real = real + b,
            .dual = dual
        };
    }

    dual_t operator - (dual_t const & b){
        return (dual_t){
            .real = real - b.real,
            .dual = dual - b.dual
        };
    }

    dual_t operator - (real_t const & b){
        return (dual_t){
            .real = real - b,
            .dual = dual
        };
    }

    dual_t operator * (dual_t const & b){
        return (dual_t) {
            .real = b.real * real,
            .dual = b.dual * real + b.real*dual
        };
    }
    dual_t operator * (real_t const & b){
        return (dual_t) {
            .real = b * real,
            .dual = b * dual
        };
    }
};

void printd(dual_t y){
    printf("%f + %f\n",y.real,y.dual);
}

dual_t operator "" _d (long double a){
    return {0.0, a};
}

/* Globally overload left-hand side operations */
static dual_t operator + (real_t a, dual_t b){ return b + a;}
static dual_t operator * (real_t a, dual_t b){ return b * a;}
static dual_t operator - (real_t a, dual_t b){ return (-1.0)*b + a;}

