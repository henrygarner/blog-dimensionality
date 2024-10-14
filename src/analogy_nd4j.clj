(ns blog-dimensionality.analogy-nd4j
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.ops.transforms Transforms]))

(defn load-data
  [file n-lines]
  (let [[mat vocab ivocab] (with-open [rdr (io/reader file)]
                             (->> (line-seq rdr)
                                  (take n-lines)
                                  (map-indexed vector)
                                  (reduce (fn [[mat vocab ivocab] [i line]]
                                            (let [segments (str/split line #" ")
                                                  word (first segments)
                                                  v (->> (rest segments)
                                                         (mapv Float/parseFloat)
                                                         (float-array))]
                                              (vector (if mat
                                                        (Nd4j/concat 0 (into-array [mat (Nd4j/create (into-array [v]))]))
                                                        (Nd4j/create (into-array [v])))
                                                      (assoc! vocab word i)
                                                      (assoc! ivocab i word))))
                                          (vector nil (transient {}) (transient {})))))
        d (Transforms/sqrt (Nd4j/sum (Transforms/pow mat 2) 0))
        W-norm (.div mat ^INDArray d)]
    {:W-norm W-norm
     :vocab (persistent! vocab)
     :ivocab (persistent! ivocab)}))


(defn analogy
  [{:keys [W-norm vocab ivocab]} a b c]
  (let [v1 (.getRow W-norm (get vocab a))
        v2 (.getRow W-norm (get vocab b))
        v3 (.getRow W-norm (get vocab c))
        vec-result (.reshape (.add v3 (.sub v2 v1)) (int-array [1 300]))
        d (Math/sqrt (.sumNumber (Transforms/pow vec-result 2)))
        vec-norm (.div vec-result d)
        similarity (.mmul W-norm (.transpose vec-norm))
        max-similarity (.sumNumber (.argMax similarity (int-array 0)))]
    (get ivocab max-similarity)))


(comment
  (defonce data (load-data "resources/glove.42B.300d.txt" 10000))

  (analogy data "mother" "mom" "father")
  ;; => dad
  )






