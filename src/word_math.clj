(ns blog-dimensionality.word-math
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [tech.v3.tensor :as t]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.argops :as op]
            [tech.v3.tensor.dimensions :as dim]
            [tech.v3.datatype.functional.opt :as opt]
            [tech.v3.parallel.for :as pfor]
            [clj-commons.primitive-math :as pmath]
            [clj-async-profiler.core :as prof]))

(defn load-data
  [file n-lines]
  (let [[n vocab ivocab] (with-open [rdr (io/reader file)]
                           (reduce (fn [[i vocab ivocab] line]
                                     (let [word (first (str/split line #" " 2))]
                                       (vector (inc i)
                                               (assoc! vocab word i)
                                               (assoc! ivocab i word))))
                                   (vector 0 (transient {}) (transient {}))
                                   (take n-lines (line-seq rdr))))
        W (t/new-tensor [n 300] {:datatype :float32})
        _ (with-open [rdr (io/reader file)]
            (doseq [[i line] (map-indexed vector (take n-lines (line-seq rdr)))
                    :let [v (->> (str/split line #" ")
                                 (rest)
                                 (mapv Float/parseFloat))]]
              (t/mset! W i v)))
        d (dfn/pow (t/reduce-axis (dfn/pow W 2) dfn/sum 1) 0.5)
        W-norm (t/transpose
                (dfn// (t/transpose W [1 0])
                       (t/broadcast d [300 n]))
                [1 0])]
    {:W-norm W-norm
     :vocab (persistent! vocab)
     :ivocab (persistent! ivocab)}))


(defn analogy
  [{:keys [W-norm vocab ivocab]} a b c]
  (let [v1 (t/mget W-norm (get vocab a))
        v2 (t/mget W-norm (get vocab b))
        v3 (t/mget W-norm (get vocab c))
        vec-result (dfn/+ v3 (dfn/- v2 v1))
        d (dfn/pow (dfn/sum (dfn/pow vec-result 2)) 0.5)
        vec-norm (dfn// vec-result d)
        shape (dim/shape (t/tensor->dimensions W-norm))
        similarity (t/reduce-axis (dfn/* W-norm (t/broadcast vec-norm shape)) dfn/sum 1)
        max-similarity (op/argmax similarity)]
    (get ivocab max-similarity)))

(comment
  (def data (load-data "resources/glove.42B.300d.txt" 10000))

  (analogy "mother" "mom" "father")
  ;; => dad
  )
