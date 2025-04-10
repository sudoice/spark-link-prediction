name := "link-prediction"

version := "1.0"

scalaVersion := "2.12.18"

val sparkVersion = "3.5.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-graphx" % sparkVersion
  
)

// Enable forking to apply javaOptions
ThisBuild / fork := true

// JVM options to allow access to internal modules
ThisBuild / javaOptions ++= Seq(
  // "-Xmx8g",
  // "-XX:+UseG1GC",
  // "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  // "--add-opens=java.base/java.util=ALL-UNNAMED",
  // "--add-opens=java.base/java.lang=ALL-UNNAMED",
  // "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  // "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED"
)