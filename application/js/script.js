// Ce scripts contient toutes les modifications côté client du site web déclencher par les boutons.
// L'application web utilise une requête ajax pour envoyer une requête à L'API et donc le modèle NLP en conséquence.

// Fonction qui modifie la couleur et le texte des élements HTML de l'application web selon le retour de la requête à l'API.

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

// Fonction qui modifie la partie détails de l'application web (mise à jour des valeurs et des barres).

function updateDetails(truevalue,falsevalue){
  $("#truebar").attr("aria-valuenow", truevalue)
  $("#falsebar").attr("aria-valuenow", falsevalue)
  $("#truebar").attr("style", "width: "+ truevalue+"%")
  $("#falsebar").attr("style", "width: "+ falsevalue+"%")
  $("#truefinfo").text(truevalue+"%");
  $("#falsefinfo").text(falsevalue+"%");
}
// Fonction modifie les éléments HTML de l'application web quand la requête a échoué.

function requestFailed(){
  $("#textalert").text("La requête a échoué.");
  $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-primary d-flex align-items-center");
  $("#infobut").hide();
}

// Fonction qui crée l'animation de chargement de l'application web pendant la requête.
function showLoading(){
  if($('#rightalert')){
    $("#textalert").text("Traitement en cours...");
  }
  else{
    $("#rightalert").append('<div class="spinner-border text-primary" role="status" id="loadalert"><span class="visually-hidden"></span></div>');
    $("#textalert").text("Traitement en cours...");
  }
}

// Fonction qui supprime l'animation de chargement de l'application web.

function hideLoading(){
  $("#loadalert").remove();
}

// Initilisation des fonctions et des états lors de la génération de la page.

$(document).ready(function() {

  // Initialisation par défaut (première page du site web <=> présentation).

  $("#presentation").show();
  $("#modele").hide();
  $("#source").hide();
  $("#infobut").hide();

  // Initialisation par le click sur bouton presentation (première page du site web <=> présentation).

  $("#iconepresentation").click(function() {
    $("#presentation").show();
    $("#modele").hide();
    $("#source").hide();
  });

   // Initialisation par le click sur bouton modele (deuxième page <=> modele <=> application web).

  $("#iconemodele").click(function() {
    $("#modele").show();
    $("#presentation").hide();
    $("#source").hide();
  });

  // Ouverture d'un nouvel onglet du navigateur utilisé vers le lien du projet GitHub.
  $("#iconesource").click(function() {
    window.open("https://github.com/Francois-lenne/Big-data-SIAD", "_blank").focus();
    
  });

  // Procédure déclenchée lorsque un tweet/texte est soumis à l'application web.

  $('#submitbut').click(function() {

    // Récupération du texte dans le champs.

    var tweet = $("#tweettext").val();

    // Vérification si le champs n'est pas vide.

    if(tweet === ""){
    
    // Si vide modification des éléments HTML en conséquence avec message.
    
    $("#textalert").text("Le champs est vide");
    $("#boxalert").removeClass($("#boxalert").attr('class')).addClass("alert alert-warning d-flex align-items-center");
    $("#infobut").hide();

    }

    // Si non vide animation de chargement et déclenchement de la requête ajax.
    else{
    showLoading();
    
    // Requête ajax
    // url -> fixé en local à modifier si l'API est déployée sur un serveur
    // type -> POST
    // data -> transformation du texte en fichier JSON
    $.ajax({
    url: "http://localhost:8000/check",
    type: "POST",
    data: JSON.stringify({ "text": tweet}),
    contentType: "application/json; charset=utf-8",
    dataType: "json",
    // Si succès déclenchement des fonctions prévus pour MAJ des élements HTML.
    success: function (data) {
        hideLoading();
        updateAlert(data['resultat'][0]);
        updateDetails(data['truevalue'],data['falsevalue']);
    },
    // Si échec déclenchement des fonctions prévus pour MAJ des élements HTML.
    error: function () {
        hideLoading();
        requestFailed();
      }
    });

    }
  });


});






  
  

