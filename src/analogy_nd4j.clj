(ns blog-dimensionality.analogy-nd4j
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.ops.transforms Transforms]))

(defn load-data
  [file n-lines]
  (let [[n vocab ivocab] (with-open [rdr (io/reader file)]
                           (->> (line-seq rdr)
                                (take n-lines)
                                (reduce (fn [[i vocab ivocab] line]
                                          (let [segments (str/split line #" " 2)
                                                word (first segments)]
                                            (vector (inc i)
                                                    (assoc! vocab word i)
                                                    (assoc! ivocab i word))))
                                        (vector 0 (transient {}) (transient {})))))
        W (Nd4j/zeros n 300 \c)
        _ (with-open [rdr (io/reader file)]
            (doseq [[i line] (->> rdr
                                  line-seq
                                  (take n-lines)
                                  (map-indexed vector))
                    :let [segments (str/split line #" ")
                          word (first segments)
                          v (->> (rest segments)
                                 (mapv Float/parseFloat)
                                 (float-array)
                                 (Nd4j/create))]
                    :when (not= word "<unk>")]
              (.putRow ^INDArray W (int i) ^INDArray v)))
        d (Transforms/sqrt (Nd4j/sum (Transforms/pow W 2) 1))
        W-norm (.transpose (.div (.transpose W) ^INDArray d))]
    {:W-norm W-norm
     :vocab (persistent! vocab)
     :ivocab (persistent! ivocab)}))


(defn analogy
  [{:keys [W-norm vocab ivocab]} a b c]
  (let [i3 (get vocab c)
        v1 (.getRow W-norm (get vocab a))
        v2 (.getRow W-norm (get vocab b))
        v3 (.getRow W-norm i3)
        vec-result (.reshape (.add v3 (.sub v2 v1)) (int-array [1 300]))
        d (Math/sqrt (.sumNumber (Transforms/pow vec-result 2)))
        vec-norm (.div vec-result d)
        similarity (.mmul W-norm (.transpose vec-norm))
        _ (.put similarity i3 0 Float/NEGATIVE_INFINITY)
        max-similarity (.sumNumber (.argMax similarity (int-array [0])))]
    (get ivocab max-similarity)))


(comment
  (def data (load-data "resources/glove.42B.300d.txt" 1e12))

  (analogy data "mother" "mom" "father")
  ;; => dad

  (analogy data "one" "two" "some")
  ;; => several

  (analogy data "one" "many" "person")
  ;; => people

  (analogy data "good" "best" "sad")
  ;; => saddest

  (analogy data "good" "best" "awful")
  ;; => horrible

  (analogy data "man" "woman" "king")
  ;; => queen
  )




