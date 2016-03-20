#ifndef HMM_H
#define HMM_H

/**
 * the realization of HMM algorithm
 * @author ZS (dragon_201209@126.com)
 * @date Mar 20, 2016 8:25:45 AM
 * c4.5
 */

#include <vector>
#include <fstream>
#include <iostream>

class ForwardHMM {
public:
    double pprob;                                  /* the probability */
    std::vector< std::vector<double> > alpha_vec;  /* the probability of observing the partial sequence, such that the state q is i */
    size_t T;                                      /* time sequence */
    std::vector<size_t> O_vec;                     /* observation sequence */
    void init(size_t _T);
};
void ForwardHMM::init(size_t _T) {
    T = _T;
    pprob = 0.0;
    O_vec.resize(T+1);
    alpha_vec.resize(T+1);
    for(size_t i=1; i<T+1; ++i) {
        alpha_vec[i].resize(T+1);
    }
}

class BackwardHMM {
public:
    double pprob;                                 /* the probability */
    std::vector< std::vector<double> > beta_vec;  /* the probability of observing the partial sequence, such that the state qt is i */
    size_t T;                                     /* time sequence */
    std::vector<size_t> O_vec;                    /* observation sequence */
    void init(size_t _T);
};
void BackwardHMM::init(size_t _T) {
    T = _T;
    pprob = 0.0;
    O_vec.resize(T+1);
    beta_vec.resize(T+1);
    for(size_t i=1; i<T+1; ++i) {
        beta_vec[i].resize(T+1);
    }
}

class ViterbiHMM {
public:
    double pprob;
    size_t N;                                      /* the number of possible hidden states */
    std::vector< std::vector<double> > delta_vec;  /* the highest probability path ending in state i */
    std::vector< std::vector<size_t> > psi;        /* the t-1 node of the maximum probability path*/
    size_t T;                                      /* time sequence */
    std::vector<size_t> O_vec;                     /* observation sequence */
    std::vector<size_t> Q_vec;                     /* the state sequence of the maximum probability path, indexes of the path */
    void init(size_t _T, size_t _N);
};
void ViterbiHMM::init(size_t _T, size_t _N) {
    T = _T;
    N = _N;
    O_vec.resize(T+1);
    Q_vec.resize(T+1);
    delta_vec.resize(N+1);
    psi.resize(N+1);
}

template<typename STATE, typename OBSERVATION>
class HMM {
public:
    HMM();
    HMM(size_t s_num, size_t v_num);
    void init(size_t s_num, size_t v_num);
    void loadHMM(char *filename);
    void printHMM();
    virtual ~HMM();

    // main algorithms
    void Forward(ForwardHMM *hmm);
    void Backward(BackwardHMM *hmm);
    void BaumWelch();
    void Viterbi(ViterbiHMM *hmm);
private:
    // the states and symbols
    std::vector<STATE> Q_vec;            /* set of states Q */
    size_t N;                            /* the number of hidden states */
    std::vector<OBSERVATION> V_vec;      /* set of symbols V */
    size_t M;                            /* the number of symbols */

    // the model is lamda = (A, B, pi)
    std::vector< std::vector<double> > A;  /* state transition matrix */
    std::vector< std::vector<double> > B;  /* Observation probability distribution */
    std::vector<double> pi;                /* the prior probability, the initial state distribution */
};

template<typename STATE, typename OBSERVATION>
HMM<STATE, OBSERVATION>::HMM() {
}

template<typename STATE, typename OBSERVATION>
HMM<STATE, OBSERVATION>::HMM(size_t s_num, size_t v_num) : N(s_num), M(v_num) {
    Q_vec.resize(N+1);
    V_vec.resize(M+1);

    pi.resize(N+1);

    A.resize(N+1);
    for(size_t i=1; i<N+1; ++i) {
        A[i].resize(N+1);
    }
    B.resize(N+1);
    for(size_t j=1; j<N+1; ++j) {
        B[j].resize(M+1);
    }
}

template<typename STATE, typename OBSERVATION>
HMM<STATE, OBSERVATION>::~HMM() {
    //
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::init(size_t s_num, size_t v_num) {
    N = s_num;
    M = v_num;

    Q_vec.resize(N+1);
    V_vec.resize(M+1);

    pi.resize(N+1);

    A.resize(N+1);
    for(size_t i=1; i<N+1; ++i) {
        A[i].resize(N+1);
    }
    B.resize(N+1);
    for(size_t j=1; j<N+1; ++j) {
        B[j].resize(M+1);
    }
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::loadHMM(char *filename) {
    std::fstream f(filename);
    f >> N;
    f >> M;
    init(N, M);

    size_t i, j;
    for(i=1; i<N+1; ++i) {
        f >> Q_vec[i];
    }
    for(i=1; i<M+1; ++i) {
        f >> V_vec[i];
    }

    for(i=1; i<N+1; ++i) {
        f >> pi[i];
    }
    for(i=1; i<N+1; ++i) {
        for(j=1; j<N+1; ++j) {
            f >> A[i][j];
        }
    }
    for(i=1; i<N+1; ++i) {
        for(j=1; j<M+1; ++j) {
            f >> B[i][j];
        }
    }
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::printHMM() {
    std::cout<<N<<std::endl;
    std::cout<<M<<std::endl;

    size_t i, j;
    for(i=1; i<N+1; ++i) {
        std::cout<<Q_vec[i]<<" ";
    }
    std::cout<<std::endl;
    for(i=1; i<M+1; ++i) {
        std::cout<<V_vec[i]<<" ";
    }
    std::cout<<std::endl;

    for(i=1; i<N+1; ++i) {
        std::cout<<pi[i]<<" ";
    }
    std::cout<<std::endl;
    for(i=1; i<N+1; ++i) {
        for(j=1; j<N+1; ++j) {
            std::cout<<A[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    for(i=1; i<N+1; ++i) {
        for(j=1; j<M+1; ++j) {
            std::cout<<B[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::Forward(ForwardHMM *hmm) {
    size_t i, j;    /* state indices */
    size_t t;       /* time index */

    double sum;     /* partial sum */

    /* 1. Initialization */
    for(i=1; i<=N; ++i) {
        hmm->alpha_vec[1][i] = pi[i] * B[i][hmm->O_vec[1]];
    }

    /* 2. Induction */
    for(t=1; t<hmm->T; ++t) {
        for(i=1; i<=N; ++i) {
            sum = 0.0;
            for(j=1; j<N; ++j) {
                sum += (hmm->alpha_vec[t][j]) * A[j][i];
            }
            hmm->alpha_vec[t+1][i] = sum * B[i][hmm->O_vec[t+1]];
        }
    }

    /* 3. Termination */
    hmm->pprob = 0.0;
    for(i=1; i<=N; ++i) {
        hmm->pprob += hmm->alpha_vec[hmm->T][i];
    }
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::Backward(BackwardHMM *hmm) {
    size_t i, j;    /* state indices */
    size_t t;       /* time index */

    double sum;     /* partial sum */

    /* 1. Initialization */
    for(i=1; i<=N; ++i) {
        hmm->beta_vec[hmm->T][i] = 1.0;
    }

    /* 2. Induction */
    for(t=hmm->T-1; t>=1; --t) {
        for(i=1; i<=N; ++i) {
            sum = 0.0;
            for(j=1; j<=N; ++j) {
                sum += A[i][j] * B[j][hmm->O_vec[t+1]] * hmm->beta_vec[t+1][j];
            }
            hmm->beta_vec[t][i] = sum;
        }
    }

    /* 3. Termination */
    hmm->pprob = 0.0;
    for(i=1; i<=N; ++i) {
        hmm->pprob += pi[i] * B[i][hmm->O_vec[1]] * (hmm->beta_vec[1][i]);
    }
}

template<typename STATE, typename OBSERVATION>
void HMM<STATE, OBSERVATION>::Viterbi(ViterbiHMM *hmm) {
    size_t i, j;       /* state indices */
    size_t t;          /* time index */

    size_t maxval_ind;

    double maxval, val;

    /* 1. Initialization  */
    for(i=1; i<N; ++i) {
        hmm->delta_vec[1][i] = pi[i] * B[i][hmm->O_vec[1]];
        hmm->psi[1][i] = 0;
    }

    /* 2. Recursion */
    for(t=2; t<=hmm->T; ++t) {
        for(i=1; i<=N; ++i) {
            maxval = 0.0;
            maxval_ind = 1;
            for(j=1; j<=N; ++j) {
                val = hmm->delta_vec[t-1][j] * A[j][i];
                if(val > maxval) {
                    maxval = val;
                    maxval_ind = j;
                }
            }

            hmm->delta_vec[t][i] = maxval * B[i][hmm->O_vec[t]];
            hmm->psi[t][i] = maxval_ind;
        }
    }

    /* 3. Termination */
    hmm->pprob = 0.0;
    hmm->Q_vec[hmm->T] = 1;
    for(i=1; i<=N; ++i) {
        if(hmm->delta_vec[hmm->T][i] > hmm->pprob) {
            hmm->pprob = hmm->delta_vec[hmm->T][i];
            hmm->Q_vec[hmm->T] = i;
        }
    }

    /* 4. Path (state sequence) backtracking */
    for(t=hmm->T-1; t>=1; --t) {
        hmm->Q_vec[t] = hmm->psi[t+1][hmm->Q_vec[t+1]];
    }
}


#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

