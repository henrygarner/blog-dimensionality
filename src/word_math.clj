(ns blog-dimensionality.word-math
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

(defn line->kv
  [line]
  (let [[word & embedding] (str/split line #" ")]
    [word (map Double/parseDouble embedding)]))

(defn load-data
  [file]
  (with-open [rdr (io/reader file)]
    (into {} (map line->kv) (take 5 (line-seq rdr)))))

(comment
  (load-data "resources/glove.42B.300d.txt"))
