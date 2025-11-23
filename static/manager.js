$(document).ready(function() {
  function loadSubreddits() {
    $.get("/get_subreddits", function(data) {
      let html = "";
      for (const [category, subs] of Object.entries(data)) {
        html += `<h5 class="mt-4">${category}</h5><ul class="list-group">`;
        if (subs.length === 0) {
          html += `<li class="list-group-item text-muted">No subreddits added yet.</li>`;
        } else {
          subs.forEach(sub => {
            html += `
              <li class="list-group-item d-flex justify-content-between align-items-center">
                r/${sub}
                <button class="btn btn-sm btn-danger remove-btn" data-category="${category}" data-subreddit="${sub}">
                  Remove
                </button>
              </li>`;
          });
        }
        html += "</ul>";
      }
      $("#subredditList").html(html);
    });
  }

  $("#addForm").submit(function(e) {
    e.preventDefault();
    $.post("/add_subreddit", $(this).serialize(), function(response) {
      $("#message").removeClass("d-none alert-success alert-danger")
                   .addClass(response.status === "success" ? "alert-success" : "alert-danger")
                   .text(response.message);
      loadSubreddits();
      $("#addForm")[0].reset();
    });
  });

  $(document).on("click", ".remove-btn", function() {
    const category = $(this).data("category");
    const subreddit = $(this).data("subreddit");
    $.post("/remove_subreddit", { category, subreddit }, function(response) {
      $("#message").removeClass("d-none alert-success alert-danger")
                   .addClass(response.status === "success" ? "alert-success" : "alert-danger")
                   .text(response.message);
      loadSubreddits();
    });
  });

  loadSubreddits();
});
