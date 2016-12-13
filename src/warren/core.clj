(ns warren.core
  (:gen-class)
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.operators :as mop]))

(mat/set-current-implementation :vectorz)

(defn rand-vec [n x]
  (take n (repeatedly #(rand x))))

(defn make-nnet [layers]
  {:bias (concat (rand-vec (dec (count layers)) 1) [0])
   :weights (loop [ls layers
                   weights []]
              (if (empty? (rest ls))
                weights
                (recur (rest ls)
                       (conj weights
                             (let [m (first ls)
                                   n (first (rest ls))]
                               (if (= n 1)
                                 (rand-vec m 1)
                                 (take m (repeatedly #(rand-vec n 1)))))))))})

(defn tanh [x]
  (Math/tanh x))

(defn dtanh [y]
  (mop/- 1.0 (mop/* y y)))

(defn sigmoid [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn dsigmoid [x]
  (mop/* x (mop/- 1 x)))

(defn mmap [f x]
  (if (vector? x)
    (mapv f x)
    (f x)))

(defn activate [inputs weights bias activation-fn]
  (mmap activation-fn
        (mop/+ bias
               (mat/mmul (mat/transpose weights) inputs))))

(defn feed [{:keys [bias weights]} inputs activation-fn]
  (reduce #(activate %1 (second %2) (first %2) activation-fn)
          inputs
          (map vector bias weights)))

(defn deltas [targets outputs dactivation-fn]
  (mop/* (mmap dactivation-fn outputs)
         (mat/sub targets outputs)))

(defn update-weights [deltas outputs weights eta]
  (mop/+ weights (mmap #(mop/* deltas eta %) outputs)))

(defn mid-deltas [deltas outputs weights dactivation-fn]
  (mop/* (mapv #(if (vector? %)
                  (reduce + %)
                  %)
               (mop/* deltas weights))
         (mapv dactivation-fn outputs)))

(defn error [outputs targets]
  (let [err (mop/- targets outputs)]
    (mop/* 0.5 (mop/* err err))))

(defn train [{:keys [bias weights]} inputs targets eta]
  (let [mid (activate inputs (first weights) (first bias) sigmoid)
        outputs (activate mid (second weights) (second bias) sigmoid)
        ;; total-error (reduce + (error outputs targets))
        o-ds (deltas targets outputs dsigmoid)
        new-o-weights (update-weights o-ds mid (second weights) eta)
        i-ds (mid-deltas o-ds mid (second weights) dsigmoid)
        new-i-weights (update-weights i-ds inputs (first weights) eta)]
    {:bias bias
     :weights [new-i-weights new-o-weights]}))

(defn cycle-train [nnet dataset times eta]
  (loop [net nnet
         n times]
    (if (zero? n)
      net
      (recur (reduce #(train %1 (first %2) (second %2) eta) net (shuffle dataset))
             (dec n)))))

(def xor-dataset [[[1 1] 0]
                  [[0 0] 0]
                  [[1 0] 1]
                  [[0 1] 1]])

(let [nnet (-> (make-nnet [2 5 1])
               (cycle-train xor-dataset 5000 0.5))]
  [(feed nnet [1 0] sigmoid)
   (feed nnet [0 1] sigmoid)
   (feed nnet [1 1] sigmoid)
   (feed nnet [0 0] sigmoid)])
