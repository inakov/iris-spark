package iris


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{Vector, DenseVector, squaredDistance}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
/**
 * Created by inakov on 16-1-20.
 */
object ScalaApp {

  def parseVector(line: String): Vector[Double] = {
    DenseVector(line.split(',').map(_.toDouble))
  }

  def closestPoint(p: Vector[Double], centers: Array[Vector[Double]]): Int = {
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 0 until centers.length) {
      val tempDist = squaredDistance(p, centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }

    bestIndex
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[2]").set("spark.executor.memory","1g")
    val sc = new SparkContext(conf)

    val lines = sc.textFile("iris.data")
    val data = lines.map(parseVector _).cache()

    val split = data.randomSplit(Array(0.2, 0.8), 1234)
    val testData = split(0).map(v => LabeledPoint(v(-1), Vectors.dense(v(0 to 3).toArray)))
    val trainData = split(1).map(v => v(0 to 3).toVector)

//    testData.foreach(println(_))
//    println("\n\n\n\n\n\n")
//    trainData.foreach(println(_))
    val K = 3
    val convergeDist = 0.01

    val kPoints = trainData.toArray()
    var tempDist = 1.0

    while(tempDist > convergeDist) {
      val closest = trainData.map (p => (closestPoint(p, kPoints), (p, 1)))

      val pointStats = closest.reduceByKey{case ((p1, c1), (p2, c2)) => (p1 + p2, c1 + c2)}

      val newPoints = pointStats.map {pair =>
        (pair._1, pair._2._1 * (1.0 / pair._2._2))}.collectAsMap()

      tempDist = 0.0
      for (i <- 0 until K) {
        tempDist += squaredDistance(kPoints(i), newPoints(i))
      }

      for (newP <- newPoints) {
        kPoints(newP._1) = newP._2
      }
      println("Finished iteration (delta = " + tempDist + ")")
    }

    println("Final centers:")
    kPoints.foreach(println)
    sc.stop()


  }

}