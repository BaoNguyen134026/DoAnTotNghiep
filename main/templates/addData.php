<?php
// doc du lieu tu website gui ve
$hoten = $_POST["ht"];
$nsinh = $_POST["ns"];
$gtinh = $_POST["gt"];

// ket noi database
include("config.php");

// gui data xuong database
$sql = "insert into bmi(hoten,nsinh,gtinh) values ('$hoten','$nsinh','$gtinh')";
mysqli_query($conn, $sql);

// ngat ket noi voi database
mysqli_close($conn);
?>