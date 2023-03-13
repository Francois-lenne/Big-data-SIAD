

function updateAlert(result){

  if(result === 1){
    $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-success d-flex align-items-center");
    $("#infobut").removeClass($("#infobut").attr('class')).addClass("btn btn-success dropdown-toggle");
    $("#textalert").text("Le tweet concerne un évênement qui est réel.");
    $("#infobut").show();
  }else if (result === 0){
    $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-danger d-flex align-items-center");
    $("#infobut").removeClass($("#infobut").attr('class')).addClass("btn btn-danger dropdown-toggle");
    $("#textalert").text("Le tweet concerne un évênement qui n'est pas réel.");
    $("#infobut").show();
    }
  
  }  

function updateDetails(truevalue,falsevalue){
  $("#truebar").attr("aria-valuenow", truevalue)
  $("#falsebar").attr("aria-valuenow", falsevalue)
  $("#truebar").attr("style", "width: "+ truevalue+"%")
  $("#falsebar").attr("style", "width: "+ falsevalue+"%")
  $("#truefinfo").text(truevalue+"%");
  $("#falsefinfo").text(falsevalue+"%");
}

function requestFailed(){
  $("#textalert").text("La requête a échoué.");
  $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-primary d-flex align-items-center");
  $("#infobut").hide();
}

function showLoading(){
  if($('#rightalert')){
    $("#textalert").text("Traitement en cours...");
  }
  else{
    $("#rightalert").append('<div class="spinner-border text-primary" role="status" id="loadalert"><span class="visually-hidden"></span></div>');
    $("#textalert").text("Traitement en cours...");
  }
}

function hideLoading(){
  $("#loadalert").remove();
}

$(document).ready(function() {


  $("#presentation").show();
  $("#modele").hide();
  $("#source").hide();
  $("#infobut").hide();



  $("#iconepresentation").click(function() {
    $("#presentation").show();
    $("#modele").hide();
    $("#source").hide();
  });

  $("#iconemodele").click(function() {
    $("#modele").show();
    $("#presentation").hide();
    $("#source").hide();
  });

  $("#iconesource").click(function() {
    window.open("https://github.com/Francois-lenne/Big-data-SIAD", "_blank").focus();
    
  });

  $('#submitbut').click(function() {

    var tweet = $("#tweettext").val();

    
    if(tweet === ""){

    $("#textalert").text("Le champs est vide");
    $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-warning d-flex align-items-center");
    $("#infobut").hide();

    }
    else{
    showLoading();
    console.log(tweet)
    

  $.ajax({
   url: "http://localhost:8000/check",
   type: "POST",
   data: JSON.stringify({ "text": tweet}),
   contentType: "application/json; charset=utf-8",
   dataType: "json",
   success: function (data) {
      console.log(data['resultat'][0])
      hideLoading();
      updateAlert(data['resultat'][0]);
      updateDetails(data['truevalue'],data['falsevalue']);
   },
   error: function () {
      hideLoading();
      requestFailed();
    }
});

    }
  });


});






  
  

