
  $('#sendbut').click(function() {

  	var input1 = $("#var1").val();
    var input2 = $("#var2").val();

  	$.ajax({
   url: "http://localhost:8000/check",
   type: "POST",
   data: JSON.stringify({ "var1": input1, "var2": input2 }),
   contentType: "application/json; charset=utf-8",
   dataType: "json",
   success: function (data) {
      console.log(data);
   }
});

  });

