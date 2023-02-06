$(document).ready(function() {
  $('#sendbut').click(function() {
    var var1 = $('#var1').val();
    var var2 = $('#var2').val();
    
    var data = { "var1": 1, "var2": 2};
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "http://localhost/predictjson", true);
	xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
	xhr.onreadystatechange = function () {
	  if (xhr.readyState === 4 && xhr.status === 200) {
	    var jsonResponse = JSON.parse(xhr.responseText);
	    console.log(jsonResponse)
	  }
	};
	xhr.send(JSON.stringify(data));



	});
  });

