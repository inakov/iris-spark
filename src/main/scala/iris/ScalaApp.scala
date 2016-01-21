package iris


import breeze.linalg.{Vector, DenseVector, squaredDistance}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.util.Random
/**
 * Created by inakov on 16-1-20.
 */
object ScalaApp {

  val K = 3
  val convergeDist = 0.00001
  val rand = new Random(42)

  def parseVector(line: String): Vector[Double] = {
    DenseVector(line.split(',').map(_.toDouble))
  }

  def closestPoint(p: Vector[Double], centers: HashMap[Int, Vector[Double]]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 1 to centers.size) {
      val vCurr = centers.get(i).get
      val tempDist = squaredDistance(p, vCurr)
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
    val data = lines.map(parseVector _)

    val split = data.randomSplit(Array(0.33, 0.67), 1234)
    val testData = split(0).map(v => (v(-1), v(0 to 3).toVector)).groupBy(_._1).map { case (k,v) => (k,v.map(_._2))}
    val trainData = split(1).map(v => v(0 to 3).toVector).toArray

    val points = new HashSet[Vector[Double]]
    val kPoints = new HashMap[Int, Vector[Double]]
    var tempDist = 1.0

    while (points.size < K) {
      points.add(trainData(rand.nextInt(trainData.length)))
    }

    val iter = points.iterator
    for (i <- 1 to points.size) {
      kPoints.put(i, iter.next())
    }

    while(tempDist > convergeDist) {
      val closest = trainData.map (p => (closestPoint(p, kPoints), (p, 1)))

      val mappings = closest.groupBy(x => x._1)

      val pointStats = mappings.map { pair =>
        pair._2.reduceLeft [(Int, (Vector[Double], Int))] {
          case ((id1, (p1, c1)), (id2, (p2, c2))) => (id1, (p1 + p2, c1 + c2))
        }
      }

      val newPoints = pointStats.map {mapping =>
        (mapping._1, mapping._2._1 * (1.0 / mapping._2._2))}

      tempDist = 0.0
      for (mapping <- newPoints) {
        tempDist += squaredDistance(kPoints.get(mapping._1).get, mapping._2)
      }

      for (newP <- newPoints) {
        kPoints.put(newP._1, newP._2)
      }
    }

    println("final centers: " + kPoints)
    testData.map(value => value._2.map(point => closestPoint(point, kPoints))).foreach(println(_))
  }

}